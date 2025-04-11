// src/lib/DatabaseAdapter.ts
import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

import { BaseRecord } from '../interfaces/baseRecord.types';

// Define a type for data being inserted (doesn't have an ID yet)
type InputData = Record<string, any>;

export class DatabaseAdapter {
    // Use 'private' keyword for encapsulation
    private readonly dbPath: string;

    /**
     * Creates an instance of DatabaseAdapter.
     * Ensures the database directory exists.
     * @param {string} [dbDirectoryName='dataBase'] - The name of the directory to use as the database root.
     */
    constructor(dbDirectoryName: string = 'dataBase') {
        this.dbPath = path.resolve(process.cwd(), dbDirectoryName);
        // Fire-and-forget initialization (could be awaited if needed before first use)
        this._initialize();
    }

    /**
     * Initializes the adapter by ensuring the database directory exists.
     * @private
     */
    private async _initialize(): Promise<void> {
        try {
            await fs.mkdir(this.dbPath, { recursive: true });
            // Optional: console.log(`Database directory ensured at: ${this.dbPath}`);
        } catch (error) {
            console.error(`Error ensuring database directory "${this.dbPath}":`, error);
            // Rethrow or handle as appropriate for your application context
            throw new Error(`Failed to initialize database directory: ${ (error as Error).message }`);
        }
    }

    /**
     * Constructs the full file path for a given table name.
     * @param {string} tableName - The name of the table (becomes the JSON filename).
     * @returns {string} The absolute path to the table's JSON file.
     * @private
     */
    private _getFilePath(tableName: string): string {
        if (!tableName || typeof tableName !== 'string' || tableName.includes('/') || tableName.includes('\\') || tableName.includes('..')) {
            throw new Error(`Invalid table name provided: "${tableName}". Must be a valid filename component.`);
        }
        return path.join(this.dbPath, `${tableName}.json`);
    }

    /**
     * Reads the entire content of a table (JSON file).
     * Returns raw data, validation happens in public methods.
     * @param {string} tableName - The name of the table.
     * @returns {Promise<unknown[]>} A promise resolving to an array of records (or empty).
     * @private
     */
    private async _readTable(tableName: string): Promise<unknown[]> {
        const filePath = this._getFilePath(tableName);
        try {
            const fileContent = await fs.readFile(filePath, 'utf-8');
            if (fileContent.trim() === '') {
                return []; // Handle empty file
            }
            // Parse JSON, but the result is initially unknown[]
            const data: unknown = JSON.parse(fileContent);
            if (!Array.isArray(data)) {
                 console.warn(`Warning: Data in table "${tableName}" is not an array. Returning empty array.`);
                 return [];
            }
            return data; // Return as unknown array
        } catch (error: unknown) {
            // If file doesn't exist, treat it as an empty table
            if (error && typeof error === 'object' && (error as NodeJS.ErrnoException).code === 'ENOENT') {
                return [];
            }
            // Log other errors (like invalid JSON) but still return empty array for resilience
            console.error(`Error reading or parsing table "${tableName}" at ${filePath}:`, error);
            return []; // Or re-throw if stricter error handling is needed
        }
    }

    /**
     * Writes the entire data array to a table's JSON file.
     * @param {string} tableName - The name of the table.
     * @param {unknown[]} data - The array of records to write.
     * @returns {Promise<void>}
     * @private
     */
    private async _writeTable(tableName: string, data: unknown[]): Promise<void> {
        const filePath = this._getFilePath(tableName);
        try {
            await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
        } catch (error: unknown) {
            console.error(`Error writing table "${tableName}" at ${filePath}:`, error);
            throw error; // Re-throw write errors
        }
    }

    // --- Public Methods ---

    /**
     * Inserts a new record (JSON object) into the specified table.
     * Generates a unique ID for the record.
     * @param {string} tableName - The name of the table (file name without .json).
     * @param {InputData} data - The data object to insert (must not contain an 'id' property).
     * @returns {Promise<BaseRecord>} A promise resolving to the inserted object, including its generated 'id'.
     */
    public async insert(tableName: string, data: InputData): Promise<BaseRecord> {
        if (typeof data !== 'object' || data === null || Array.isArray(data)) {
            throw new Error('Data to insert must be an object.');
        }
        if (data.hasOwnProperty('id')) {
            throw new Error("Input data should not contain an 'id' property; it will be generated.");
        }

        const tableData = await this._readTable(tableName);

        const newRecord: BaseRecord = {
            ...data,
            id: crypto.randomUUID(), // Generate unique ID
        };

        // Add the new record to the array
        tableData.push(newRecord);

        await this._writeTable(tableName, tableData);
        return newRecord;
    }

    /**
     * Inserts multiple records into the specified table.
     * Generates a unique ID for each record.
     * @param {string} tableName - The name of the table.
     * @param {InputData[]} dataArray - An array of data objects to insert.
     * @returns {Promise<BaseRecord[]>} A promise resolving to the array of inserted objects, including their generated 'id'.
     */
    public async insertMany(tableName: string, dataArray: InputData[]): Promise<BaseRecord[]> {
        if (!Array.isArray(dataArray)) {
            throw new Error('Input data must be an array of objects.');
        }

        const tableData = await this._readTable(tableName);
        const newRecords: BaseRecord[] = [];

        for (const data of dataArray) {
            if (typeof data !== 'object' || data === null || Array.isArray(data)) {
                console.warn('Skipping invalid item in batch insert:', data);
                continue; // Or throw error if strictness is needed
            }
            if (data.hasOwnProperty('id')) {
                 console.warn(`Skipping item with existing 'id' property in batch insert: ${data.id}`);
                continue; // Or throw error
            }
            const newRecord: BaseRecord = {
                ...data,
                id: crypto.randomUUID(),
            };
            tableData.push(newRecord);
            newRecords.push(newRecord);
        }

        if (newRecords.length > 0) {
            await this._writeTable(tableName, tableData);
        }

        return newRecords;
    }

    /**
     * Retrieves a single record from a table by its ID.
     * @param {string} tableName - The name of the table.
     * @param {string} id - The unique ID of the record to retrieve.
     * @returns {Promise<BaseRecord | null>} A promise resolving to the found record object, or null if not found or invalid.
     */
    public async getById(tableName: string, id: string): Promise<BaseRecord | null> {
        if (!id || typeof id !== 'string') {
            return null; // Invalid ID provided
        }
        const tableData = await this._readTable(tableName);

        // Find and validate the record
        const record = tableData.find((r: unknown): r is BaseRecord =>
            r !== null && typeof r === 'object' && 'id' in r && r.id === id
        );

        return record || null;
    }

    /**
     * Retrieves all valid records from a specified table.
     * Filters out any entries that don't conform to the BaseRecord structure (having an ID).
     * @param {string} tableName - The name of the table.
     * @returns {Promise<BaseRecord[]>} A promise resolving to an array containing all valid records in the table.
     */
    public async getAll(tableName: string): Promise<BaseRecord[]> {
        const tableData = await this._readTable(tableName);

        // Filter and type guard to ensure records have the expected shape
        const validRecords = tableData.filter((r: unknown): r is BaseRecord =>
            r !== null && typeof r === 'object' && 'id' in r && typeof r.id === 'string'
        );

        return validRecords;
    }
}

export default DatabaseAdapter; // Export the class
